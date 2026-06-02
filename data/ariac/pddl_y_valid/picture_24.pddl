(define (problem picture_24)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_regulator red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_regulator)
        (clear green_regulator)
        (part_at blue_battery table)
        (part_at red_regulator table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at green_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
