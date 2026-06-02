(define (problem picture_4)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_regulator green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_regulator)
        (clear red_regulator)
        (clear blue_battery)
        (part_at blue_battery table)
        (part_at red_regulator table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
