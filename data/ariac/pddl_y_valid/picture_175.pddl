(define (problem picture_175)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_battery blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear blue_battery)
        (part_at green_battery table)
        (part_at red_battery table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and
            (part_at green_battery battery_placement)
        )
    )
)
