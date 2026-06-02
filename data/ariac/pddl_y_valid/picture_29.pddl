(define (problem picture_29)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_regulator blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_regulator)
        (clear red_battery)
        (clear blue_battery)
        (part_at green_regulator table)
        (part_at red_battery table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and
            (part_at green_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
